package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Font;
import com.itextpdf.text.FontFactory;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.PdfWriter;


public class SetParagraphFontSize {
			
	
	public static void main(String[] args){
		Document document = new Document();// 创建文档对象
		try {
			PdfWriter.getInstance(document, new FileOutputStream("c:\\个人简历.pdf"));// 关联文档对象与输出流
			document.open();// 打开文档
			document.add(new Paragraph("Personal Resumes",FontFactory.getFont(FontFactory.HELVETICA,50,Font.BOLDITALIC)));// 定义段落字体属性并添加内容
			document.close();// 关闭文档
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (DocumentException e) {
			e.printStackTrace();
		}
	}
}
