package com.zzk;

import java.io.File;
import java.io.FileInputStream;
import java.net.MalformedURLException;
import org.apache.pdfbox.pdfparser.PDFParser;
import org.apache.pdfbox.pdmodel.PDDocument;
import org.apache.pdfbox.util.PDFTextStripper;
import com.itextpdf.text.Document;

public class ReaderPdfFile {

	public static void main(String[] args) throws MalformedURLException {
		Document document = new Document();// 创建文档对象
		File file = new File("c:\\创建第一个PDF文档.pdf");// 创建File对象
		try {
			FileInputStream in = new FileInputStream(file);// 创建输入流对象
			PDFParser parser = new PDFParser(in);// 创建PDF解析器
			parser.parse();// 解析PDF文档
			PDDocument pdfdocument = parser.getPDDocument();// 获得解析后的PDF文档
			PDFTextStripper stripper = new PDFTextStripper();// 创建PDF文本剥离器
			String msg = stripper.getText(pdfdocument);// 使用剥离器从PDF文档中剥离文本信息
			System.out.println("请取到的PDF文本信息如下:\n" + msg);// 输出信息
			in.close();// 关闭输入流对象
		} catch (Exception e) {
			e.printStackTrace();
		}
		document.close();// 关闭文档
	}
}
