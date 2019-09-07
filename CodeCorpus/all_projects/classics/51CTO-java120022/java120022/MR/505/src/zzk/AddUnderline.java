package zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;

import com.itextpdf.text.Chunk;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Font;
import com.itextpdf.text.FontFactory;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.PdfWriter;

public class AddUnderline {
	
	
	public static void main(String[] args){
		Document document = new Document();// 创建文档对象
		try {
			PdfWriter.getInstance(document, new FileOutputStream("C:\\明日科技简介.pdf"));// 关联文档对象与输出流
			document.open();// 打开文档
			document.add(new Paragraph("Generalize"));// 向文档中添加内容
			document.add(new Paragraph("Welcome to GuangZhou！", FontFactory.getFont(FontFactory.HELVETICA,15,Font.UNDERLINE)));// 向文档中添加内容
			document.close();// 关闭文档
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (DocumentException e) {
			e.printStackTrace();
		}
	}
}

