package com.zzk;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import com.itextpdf.text.Document;
import com.itextpdf.text.DocumentException;
import com.itextpdf.text.Paragraph;
import com.itextpdf.text.pdf.PdfWriter;

public class AddandCreateDate {
	public static void main(String args[]){
		Document document = new Document();// 创建文档对象
		try {
			PdfWriter.getInstance(document, new FileOutputStream("c:\\Java编程词典.pdf"));// 关联文档对象与输出流
			document.addAuthor("明日科技");	//添加作者
			document.addCreationDate();// 创建日期
			document.open();// 打开文档
			document.add(new Paragraph("CreateDate"));// 向文档中添加内容
			document.close();// 关闭文档
		} catch (FileNotFoundException e) {			
			e.printStackTrace();
		} catch (DocumentException e) {
			e.printStackTrace();
		}
	}
}
